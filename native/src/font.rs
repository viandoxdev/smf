//! Main font processing / atlas generation logic
//! Notes:
//!   - There are several coordinate spaces at work here and it can get confusing very quickly
//!   without some order, so heres a few:
//!     -> Shape Space: The coordinate of the glyph's outline, coordinate are expressed in f32s
//!          (since we're working with curves here). The outline doesn't necessarily start at the origin
//!          (some parts of it can be outside), usually a tight BoundingBox<f32> around the outline will
//!          be available when working with this space. Scale has yet to be applied in at this point so
//!          everything is as it is in the font file
//!     -> Raster Space: The coordinate of the rasterized glyph, coordinate are expressed in pixels
//!          (so u32 or any other integer type), or sometime f32s when needed. Scale has been applied
//!          at this point. The glyph is positionned as tightly as possible in raster space: its top
//!          left most point will only be shifted by the glyph_padding from the origin (in both axis),
//!          and the space is no bigger than the rasterized glyph's size plus twice the padding in every
//!          direction. Coordinates are strictly positive.
//!
//! Also credits to Cosmic text are due, since this implementation roughly follows the same
//! structure

use std::{
    collections::HashMap,
    ffi::c_char,
    fmt::Display,
    io::Write,
    iter,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Range},
    str::FromStr,
    sync::Arc,
};

use ab_glyph_rasterizer::{Point, Rasterizer};
use fdsm::{
    generate::{generate_msdf, generate_sdf},
    render::{correct_sign_msdf, correct_sign_sdf},
    shape::Shape,
    transform::Transform,
};
use image::{GenericImage, GenericImageView, ImageBuffer, Luma, Pixel, Primitive, Rgb, SubImage};
use itertools::Itertools;
use nalgebra::{Affine2, Scale2, Similarity2, TAffine, Transform2, Translation2, Vector2};
use rustybuzz::{
    Direction, Face as BuzzFace, GlyphBuffer, GlyphInfo, GlyphPosition, Language, ShapePlan,
    UnicodeBuffer,
};
use ttf_parser::{fonts_in_collection, Face, FaceParsingError, GlyphId, OutlineBuilder, Tag};
use unicode_bidi::{BidiInfo, Level};
use unicode_linebreak::BreakOpportunity;
use unicode_script::{Script, ScriptExtension, UnicodeScript};

use crate::{
    config::{FontConfig, GlobalConfig},
    error::SMFError,
    packing::{BoundingBox, Heuristics, Layered, LayeredOnlinePacker, MaxRectPacker},
    textures::{Diff, GenericTextureStore, TextureStore},
};

#[derive(Debug, Clone, Copy)]
pub enum RasterKind {
    /// Bitmap rasterization (will look blurry when scaled up)
    Bitmap = 0,
    /// Signed Distance Field, will look sharp regardless of scale but will smooth up corners and
    /// makes edges slightly jagged
    SDF = 1,
    /// Multi Chanel Signed Distance Field, fixes SDF's smooth corner problem
    MSDF = 2,
}

impl TryFrom<i32> for RasterKind {
    type Error = SMFError;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Bitmap),
            1 => Ok(Self::SDF),
            2 => Ok(Self::MSDF),
            _ => Err(SMFError::MalformedRasterKind),
        }
    }
}

#[derive(Hash, Debug, PartialEq, Eq, Clone)]
struct ShapePlanConfig {
    direction: Direction,
    language: Arc<Language>,
    script: rustybuzz::Script,
}

/// Intermediary struct used in glyph rasterization
struct GlyphRasterInfo {
    /// The glyph's id
    id: GlyphId,
    /// The glyph's tight bounding box in shape space
    bbox: BoundingBox<f32>,
    /// On what layer / which texture should the raster be
    packed_layer: u32,
    /// The raster's position on such texture, in pixels
    packed: BoundingBox<u32>,
    /// The raster's width, without glyph padding
    width: u32,
    /// The raster's height, without glyph padding
    height: u32,
    /// The glyph padding
    padding: u32,
}

impl GlyphRasterInfo {
    fn padded_width(&self) -> u32 {
        self.width + 2 * self.padding
    }
    fn padded_height(&self) -> u32 {
        self.height + 2 * self.padding
    }
    fn transform(&self) -> Affine2<f64> {
        // Translate the outlines so that the top left most point is at the origin
        let origin_to_top_left_corner =
            Translation2::new(-self.bbox.x1 as f64, -self.bbox.y1 as f64);
        // Scale everything so that it fits into the raster's size
        let scale = nalgebra::convert::<_, nalgebra::Transform<f64, TAffine, 2>>(Scale2::new(
            self.width as f64 / self.bbox.width() as f64,
            self.height as f64 / self.bbox.height() as f64,
        ));
        // Translate by the padding
        let add_padding = Translation2::new(self.padding as f64, self.padding as f64);
        // Flip vertically
        let flip = Translation2::new(0.0, self.padded_height() as f64)
            * nalgebra::convert::<_, nalgebra::Transform<f64, TAffine, 2>>(Scale2::new(1.0, -1.0));

        // Transformations are applied in reverse order from multiplication
        // (origin_to_top_left_corner will be applied first, then scale, ...)
        nalgebra::convert(flip * add_padding * scale * origin_to_top_left_corner)
    }
}

fn rect_to_bbox(rect: ttf_parser::Rect) -> BoundingBox<f32> {
    BoundingBox::new(
        rect.x_min as f32,
        rect.y_min as f32,
        rect.x_max as f32,
        rect.y_max as f32,
    )
}

fn level_to_dir(lvl: Level) -> Direction {
    if lvl.is_ltr() {
        Direction::LeftToRight
    } else {
        Direction::RightToLeft
    }
}

pub struct Command {
    pub str: String,
    pub multiline: bool,
    pub max_length: Option<f32>,
}

pub struct BatchedCommands {
    pub commands: Vec<Command>,
}

pub struct BatchedResults {
    pub meshes: Vec<Result<Vec<Mesh>, SMFError>>,
    pub diffs: Vec<Diff>,
}

pub struct Mesh {
    pub texture: u32,
    pub vertices: Box<[f32]>,
    pub indices: Box<[u32]>,
}

struct ShapedGlyph {
    cluster_start: usize,
    cluster_end: usize,
    pos: GlyphPosition,
    glyph_id: GlyphId,
}

/// Here read a sequence of unbreakable glyphs
struct ShapedWord {
    glyphs: Vec<ShapedGlyph>,
    /// Whether the line should break after this words (mandatory line break)
    end_of_line: bool,
    advance: i32,
}

impl ShapedWord {
    pub fn new(str: &str, end_of_line: bool, level: Level, font: &mut Font) -> Self {
        let mut glyphs = Vec::new();
        let mut advance = 0;

        let mut scripts = ScriptExtension::default();
        let iter = str
            .char_indices()
            .map(|(i, c)| (i, c.script_extension()))
            .chain(std::iter::once((
                str.len(),
                ScriptExtension::from(Script::Unknown),
            )));
        let mut run_start = 0;
        for (i, char_scripts) in iter {
            let next_scripts = scripts.intersection(char_scripts);

            if next_scripts.is_empty() {
                let script = scripts
                    .iter()
                    .next()
                    .expect("Scripts isn't empty yet yields no script");
                let buzz_script = rustybuzz::Script::from_str(script.short_name())
                    .expect("Couldn't convert Script to rustybuzz Script");
                let cfg = ShapePlanConfig {
                    direction: level_to_dir(level),
                    language: font.config.language.clone(),
                    script: buzz_script,
                };
                let buf = font.shape(&str[run_start..i], &cfg);

                glyphs.extend(
                    buf.glyph_positions()
                        .iter()
                        .zip(buf.glyph_infos().iter())
                        .map(|(pos, info)| ShapedGlyph {
                            cluster_start: info.cluster as usize + run_start,
                            cluster_end: i,
                            pos: *pos,
                            glyph_id: GlyphId(info.glyph_id as u16),
                        }),
                );

                for GlyphPosition { x_advance, .. } in buf.glyph_positions() {
                    advance += x_advance
                }

                run_start = i;
                scripts = char_scripts;
            } else {
                scripts = next_scripts;
            }
        }

        // Set glyph cluster end
        if level.is_ltr() {
            for i in (1..glyphs.len()).rev() {
                let next_end = glyphs[i].cluster_end;
                let next_start = glyphs[i].cluster_start;
                let cur_start = glyphs[i - 1].cluster_start;
                glyphs[i - 1].cluster_end = if cur_start == next_start {
                    next_end
                } else {
                    next_start
                };
            }
        } else {
            for i in 1..glyphs.len() {
                let next_end = glyphs[i - 1].cluster_end;
                let next_start = glyphs[i - 1].cluster_start;
                let cur_start = glyphs[i].cluster_start;
                glyphs[i].cluster_end = if cur_start == next_start {
                    next_end
                } else {
                    next_start
                }
            }
        }

        Self {
            glyphs,
            end_of_line,
            advance,
        }
    }
}

/// A level run of words
struct ShapedSpan {
    words: Vec<ShapedWord>,
    level: Level,
}

impl ShapedSpan {
    pub fn new(
        str: &str,
        range: Range<usize>,
        linebreaks: &[(usize, BreakOpportunity)],
        level: Level,
        font: &mut Font,
    ) -> Self {
        let mut words = Vec::new();

        // Isolate the range of linebreaks opportunities we care about
        let lb_start = linebreaks
            .binary_search_by_key(&range.start, |(k, _)| *k)
            .unwrap_or_else(|x| x);
        let lb_end = linebreaks
            .binary_search_by_key(&range.end, |(k, _)| *k)
            .unwrap_or_else(|x| x);
        // We add an opportunity at the end of the span because it is also a word boundary
        let lb_iter = linebreaks[lb_start..lb_end]
            .iter()
            .copied()
            .chain(std::iter::once((range.end, BreakOpportunity::Allowed)));

        let mut word_start = range.start;
        for (i, op) in lb_iter {
            words.push(ShapedWord::new(
                &str[word_start..i],
                op == BreakOpportunity::Mandatory,
                level,
                font,
            ));
            word_start = i;
        }

        Self { words, level }
    }
}

struct ShapedParagraph {
    base_level: Level,
    spans: Vec<ShapedSpan>,
}

impl ShapedParagraph {
    pub fn new(str: &str, font: &mut Font) -> Vec<Self> {
        let bidi = BidiInfo::new(str, Some(Level::ltr()));
        let lb = unicode_linebreak::linebreaks(str).collect_vec();
        let mut pars = Vec::new();

        for par in bidi.paragraphs {
            let mut spans = Vec::new();
            let mut run_start = par.range.start;
            let mut run_level = bidi.levels[par.range.start];
            for i in par.range.start..=par.range.end {
                if i == bidi.levels.len() || bidi.levels[i] != run_level {
                    // End the current run
                    spans.push(ShapedSpan::new(&str, run_start..i, &lb, run_level, font));

                    if i < bidi.levels.len() {
                        run_level = bidi.levels[i];
                        run_start = i;
                    }
                }
            }

            pars.push(Self {
                base_level: par.level,
                spans,
            });
        }

        pars
    }

    pub fn consume(self, res: &mut Vec<PreLayoutGlyph>) {
        if self.spans.iter().all(|s| s.words.is_empty()) {
            return;
        }

        for span in self.spans {
            for word in span.words {
                for glyph in word.glyphs {
                    res.push(PreLayoutGlyph {
                        glyph_id: glyph.glyph_id,
                        x_advance: glyph.pos.x_advance,
                        x_offset: glyph.pos.x_offset,
                        y_offset: glyph.pos.y_offset,
                        end_of_line: false,
                        end_of_paragraph: false,
                        level: span.level,
                        base_level: self.base_level,
                    });
                }
            }
        }

        if let Some(last) = res.last_mut() {
            last.end_of_line = true;
            last.end_of_paragraph = true;
        }
    }

    pub fn consume_line_wrap(self, max_length: f32, scale: f32, res: &mut Vec<PreLayoutGlyph>) {
        let mut x = 0f32;

        if self.spans.iter().all(|s| s.words.is_empty()) {
            return;
        }

        let mut first_of_line = true;
        let mut end_line = false;
        for span in self.spans {
            for word in span.words {
                x += word.advance as f32 * scale;
                if (x >= max_length && !first_of_line) || end_line {
                    res.last_mut().unwrap().end_of_line = true;
                    x = 0.0;
                    first_of_line = true;
                } else {
                    first_of_line = false;
                }

                for glyph in word.glyphs {
                    res.push(PreLayoutGlyph {
                        glyph_id: glyph.glyph_id,
                        x_advance: glyph.pos.x_advance,
                        x_offset: glyph.pos.x_offset,
                        y_offset: glyph.pos.y_offset,
                        end_of_line: false,
                        end_of_paragraph: false,
                        level: span.level,
                        base_level: self.base_level,
                    });
                }

                // Handle mandatory line breaks
                end_line = word.end_of_line;
            }
        }

        if let Some(last) = res.last_mut() {
            last.end_of_line = true;
            last.end_of_paragraph = true;
        }
    }
}

struct PreLayoutGlyph {
    glyph_id: GlyphId,
    x_advance: i32,
    x_offset: i32,
    y_offset: i32,
    end_of_line: bool,
    end_of_paragraph: bool,
    level: Level,
    base_level: Level,
}

pub struct Font<'a, P: LayeredOnlinePacker<GlyphId> = Layered<MaxRectPacker<GlyphId>, GlyphId>> {
    shape_plan_chache: HashMap<ShapePlanConfig, ShapePlan>,
    face: BuzzFace<'a>,
    packer: P,
    pub config: FontConfig,
    gconf: GlobalConfig,
    textures: Box<dyn GenericTextureStore>,
}

impl<'f> Font<'f> {
    pub fn from_bytes<'a: 'f>(bytes: &'a [u8], config: FontConfig) -> Result<Vec<Self>, SMFError> {
        let n = fonts_in_collection(bytes).unwrap_or(1);

        let cfg = GlobalConfig::get_copy();

        // Make sure the config has been set
        if !cfg.is_set() {
            return Err(SMFError::GlobalConfigNotSet);
        }

        Ok((0..n)
            .map(|i| {
                let face = Face::parse(bytes, i)?;
                Ok(Self {
                    face: BuzzFace::from_face(face),
                    shape_plan_chache: HashMap::new(),
                    packer: Layered::new(cfg.atlas_size, cfg.atlas_size, |w, h| {
                        MaxRectPacker::new(w, h, Some(Heuristics::MinY))
                    }),
                    config: config.clone(),
                    textures: match config.raster_kind {
                        RasterKind::SDF | RasterKind::Bitmap => {
                            Box::new(TextureStore::<Luma<u8>>::new(cfg.atlas_size))
                        }
                        RasterKind::MSDF => Box::new(TextureStore::<Rgb<u8>>::new(cfg.atlas_size)),
                    },
                    gconf: cfg.clone(),
                })
            })
            .collect::<Result<Vec<Self>, FaceParsingError>>()
            .map_err(|e| SMFError::FailedParsing(Some(e)))?)
    }

    pub fn from_bytes_and_index<'a: 'f>(
        bytes: &'a [u8],
        index: u32,
        config: FontConfig,
    ) -> Result<Self, SMFError> {
        let cfg = GlobalConfig::get_copy();

        // Make sure the config has been set
        if !cfg.is_set() {
            return Err(SMFError::GlobalConfigNotSet);
        }

        let face = Face::parse(bytes, index)?;
        Ok(Self {
            face: BuzzFace::from_face(face),
            shape_plan_chache: HashMap::new(),
            packer: Layered::new(cfg.atlas_size, cfg.atlas_size, |w, h| {
                MaxRectPacker::new(w, h, Some(Heuristics::MinY))
            }),
            config: config.clone(),
            textures: match config.raster_kind {
                RasterKind::SDF | RasterKind::Bitmap => {
                    Box::new(TextureStore::<Luma<u8>>::new(cfg.atlas_size))
                }
                RasterKind::MSDF => Box::new(TextureStore::<Rgb<u8>>::new(cfg.atlas_size)),
            },
            gconf: cfg.clone(),
        })
    }

    fn scale(&self) -> f32 {
        self.config.scale / self.face.units_per_em() as f32
    }

    pub fn process_batched(&mut self, commands: BatchedCommands) -> BatchedResults {
        let meshes = commands
            .commands
            .into_iter()
            .map(|c| self.process(c))
            .collect_vec();
        let diffs = self.textures.take_concatenated_diffs();
        BatchedResults { meshes, diffs }
    }

    pub fn process(&mut self, command: Command) -> Result<Vec<Mesh>, SMFError> {
        if command.multiline {
            self.process_multiline(&command.str, command.max_length)
        } else {
            self.process_uniline(&command.str)
        }
    }

    pub fn process_uniline(&mut self, str: &str) -> Result<Vec<Mesh>, SMFError> {
        let paragraphs = ShapedParagraph::new(str, self);
        let mut pre_layout_glyphs = Vec::new();
        for par in paragraphs {
            par.consume(&mut pre_layout_glyphs);
        }

        self.layout(&mut pre_layout_glyphs)
    }

    pub fn process_multiline(
        &mut self,
        str: &str,
        max_length: Option<f32>,
    ) -> Result<Vec<Mesh>, SMFError> {
        let max_length = max_length.unwrap_or(f32::MAX);
        let paragraphs = ShapedParagraph::new(str, self);
        let mut pre_layout_glyphs = Vec::new();
        for par in paragraphs {
            par.consume_line_wrap(max_length, self.scale(), &mut pre_layout_glyphs);
        }

        self.layout(&mut pre_layout_glyphs)
    }

    // TODO: Figure language out through configurable language list and script value rather than
    // just requesting it (see: https://github.com/harfbuzz/harfbuzz/issues/1288)

    fn layout(&mut self, lines: &mut Vec<PreLayoutGlyph>) -> Result<Vec<Mesh>, SMFError> {
        // TODO: Magic constant
        let line_height = self.face.height() as f32 * 0.6 * self.scale() * self.config.line_height;

        #[derive(Clone)]
        struct BuildingMesh {
            vertices: Vec<f32>,
            indices: Vec<u32>,
            texture: u32,
            empty: bool,
        }

        impl From<BuildingMesh> for Mesh {
            fn from(value: BuildingMesh) -> Self {
                Mesh {
                    texture: value.texture,
                    vertices: value.vertices.into_boxed_slice(),
                    indices: value.indices.into_boxed_slice(),
                }
            }
        }

        impl BuildingMesh {
            #[inline(always)]
            fn add_quad(&mut self, vertices: [f32; 16]) {
                let i = self.vertices.len() as u32 / 4;
                self.vertices.extend_from_slice(&vertices);
                self.indices
                    .extend_from_slice(&[i, i + 1, i + 2, i, i + 2, i + 3]);
                self.empty = false;
            }
        }

        let mut meshes = Vec::<BuildingMesh>::new();

        let mut y = 0f32;
        let mut line_start_index = 0;
        let mut base_level = lines.first().map(|g| g.base_level).unwrap_or(Level::ltr());

        for i in 0..=lines.len() {
            // We first need to indentify the line
            if i > 0 && !lines[i - 1].end_of_line {
                continue;
            }

            // The last line ended (and we may need to start a new one)
            if base_level.is_rtl() {
                lines[line_start_index..i].reverse();
            }

            // Reverse runs of text with directions different than the base one
            // For that we first need to figure out the runs
            {
                let mut level = base_level;
                let mut run_start = line_start_index;
                for j in line_start_index..=i {
                    if j < i && lines[j].level == level {
                        // Glyph at index j is part of the current run
                        continue;
                    }

                    if level != base_level {
                        // Run ended and is of a different direction than base: reverse
                        lines[run_start..j].reverse();
                    }

                    if j < i {
                        // Start a new run
                        level = lines[j].level;
                        run_start = j;
                    }
                }
            }

            // Bidi reordering has been done, we just need to layout the line now

            let mut x = 0f32;
            let mut last_glyph = None;
            let subtables = self.face.tables().kern.map(|s| s.subtables);

            for g in &lines[line_start_index..i] {
                let loc = self.get_glyph_location(g.glyph_id)?;
                let bbox = self
                    .face
                    .glyph_bounding_box(g.glyph_id)
                    .map(rect_to_bbox)
                    .unwrap_or(BoundingBox::ZERO);

                // TODO: Is this actually doing anything ?
                let kerning = if let (Some(last_glyph), Some(subtables)) = (last_glyph, subtables) {
                    let kern = subtables
                        .into_iter()
                        .flat_map(|t| t.glyphs_kerning(last_glyph, g.glyph_id))
                        .next()
                        .unwrap_or_default();
                    kern as f32 * self.scale()
                } else {
                    0.0
                };

                last_glyph = Some(g.glyph_id);

                let (gx, gy) = (
                    x + g.x_offset as f32 * self.scale() + kerning,
                    y + g.y_offset as f32 * self.scale(),
                );

                x += g.x_advance as f32 * self.scale();

                let Some((tc, tex)) = loc else {
                    continue;
                };

                if bbox.width() == 0.0 || bbox.height() == 0.0 {
                    // No point in laying out a glyph that won't get rendered
                    continue;
                }

                let (w, h) = (bbox.width() * self.scale(), bbox.height() * self.scale());

                if tex as usize >= meshes.len() {
                    meshes.extend((meshes.len()..=tex as usize).map(|t| BuildingMesh {
                        vertices: Vec::new(),
                        indices: Vec::new(),
                        texture: t as u32,
                        empty: true,
                    }));
                }

                meshes[tex as usize].add_quad([
                    gx + w,
                    gy + h,
                    tc.x2,
                    tc.y1,
                    gx,
                    gy + h,
                    tc.x1,
                    tc.y1,
                    gx,
                    gy,
                    tc.x1,
                    tc.y2,
                    gx + w,
                    gy,
                    tc.x2,
                    tc.y2,
                ]);
            }

            // We're done with this line, start a new one
            // (Or don't if we're done with all the lines)
            if i < lines.len() {
                line_start_index = i;
                y -= line_height;
                base_level = lines[i].base_level;
            }
        }

        Ok(meshes
            .into_iter()
            .filter(|b| !b.empty)
            .map(Mesh::from)
            .collect_vec())
    }

    /// Shape a run of text
    fn shape(&mut self, str: &str, cfg: &ShapePlanConfig) -> GlyphBuffer {
        let plan = match self.shape_plan_chache.get(cfg) {
            Some(p) => p,
            None => {
                let plan = ShapePlan::new(
                    &self.face,
                    cfg.direction,
                    Some(cfg.script),
                    Some(&cfg.language),
                    &[],
                );
                self.shape_plan_chache.entry(cfg.clone()).or_insert(plan)
            }
        };

        let mut buffer = UnicodeBuffer::new();
        buffer.push_str(str);
        buffer.set_script(cfg.script);
        buffer.set_direction(cfg.direction);
        // TODO: no
        buffer.set_language(Language::clone(&*cfg.language));

        rustybuzz::shape_with_plan(&self.face, plan, buffer)
    }

    /// Get a glyph's location in the atlasses, rasterizing it if it isn't already.
    fn get_glyph_location(
        &mut self,
        glyph_id: GlyphId,
    ) -> Result<Option<(BoundingBox<f32>, u32)>, SMFError> {
        if let Some((bbox, layer)) = self.packer.get_loc(&glyph_id) {
            Ok(Some((
                bbox.map(|v| v as f32 / self.gconf.atlas_size as f32),
                layer,
            )))
        } else {
            self.rasterize_glyph(glyph_id)
        }
    }

    /// Rasterize a glyph to a SDF atlas
    fn rasterize_sdf(&mut self, glyph: GlyphRasterInfo) -> Result<(), SMFError> {
        let mut shape = Shape::load_from_face(&self.face, glyph.id);

        shape.transform(&glyph.transform());

        let tex = self
            .textures
            .get_texture_luma(glyph.packed_layer)
            .ok_or_else(|| SMFError::ExtraError("TextureStore is of wrong type".to_string()))?;

        let mut sub = tex.sub_image(
            glyph.packed.x1,
            glyph.packed.y1,
            glyph.padded_width(),
            glyph.padded_height(),
        );

        let prepared = shape.prepare();

        generate_sdf(&prepared, 4.0, &mut *sub);
        correct_sign_sdf(
            &mut *sub,
            &prepared,
            fdsm::bezier::scanline::FillRule::Nonzero,
        );

        Ok(())
    }

    /// Rasterize a glyph to a MSDF atlas
    fn rasterize_msdf(&mut self, glyph: GlyphRasterInfo) -> Result<(), SMFError> {
        let mut shape = Shape::load_from_face(&self.face, glyph.id);

        shape.transform(&glyph.transform());

        let colored_shape =
            Shape::edge_coloring_simple(shape, self.gconf.sin_alpha, self.gconf.coloring_seed);
        let prepared = colored_shape.prepare();

        let tex = self
            .textures
            .get_texture_rgb(glyph.packed_layer)
            .ok_or_else(|| SMFError::ExtraError("TextureStore is of wrong type".to_string()))?;

        let mut sub = tex.sub_image(
            glyph.packed.x1,
            glyph.packed.y1,
            glyph.packed.width(),
            glyph.packed.height(),
        );

        generate_msdf(&prepared, 4.0, &mut *sub);
        correct_sign_msdf(
            &mut *sub,
            &prepared,
            fdsm::bezier::scanline::FillRule::Nonzero,
        );

        Ok(())
    }

    /// Rasterize a glyph to a bitmap atlas
    fn rasterize_bitmap(&mut self, glyph: GlyphRasterInfo) -> Result<(), SMFError> {
        let mut builder = Builder::new(glyph.padding, glyph.bbox, glyph.width, glyph.height);
        self.face.outline_glyph(glyph.id, &mut builder);

        let tex = self
            .textures
            .get_texture_luma(glyph.packed_layer)
            .ok_or_else(|| SMFError::ExtraError("TextureStore is of wrong type".to_string()))?;

        builder.rasterizer.for_each_pixel_2d(|x, y, v| {
            tex.put_pixel(
                x + glyph.packed.x1,
                // Flip because the glyph's coordinate system has y pointing up, unlike the images'
                (glyph.packed.height() - y) + glyph.packed.y1,
                Luma([(v * 255.0) as u8]),
            );
        });

        Ok(())
    }

    /// Rasterize a glyph and add it to the atlas, returning its location
    fn rasterize_glyph(
        &mut self,
        glyph_id: GlyphId,
    ) -> Result<Option<(BoundingBox<f32>, u32)>, SMFError> {
        let Some(rect) = self.face.glyph_bounding_box(glyph_id) else {
            return if let Some(_) = self.face.glyph_svg_image(glyph_id) {
                // TODO: Handle SVG glyphs
                Err(SMFError::UnsupportedGlyphFormat)
            } else if let Some(_) = self.face.glyph_raster_image(glyph_id, 0) {
                // TODO: Handle raster glyphs
                Err(SMFError::UnsupportedGlyphFormat)
            } else {
                // NOTE: This is probably wrong but we'll catch that later
                // -> We need a way to handle missing glyphs with font fallbacks and what not
                Ok(None)
            };
        };

        let padding = self.gconf.glyph_padding;

        // Glyph bounding box, in shape space (what we'll call the outline's raw coordinate space)
        let bbox = rect_to_bbox(rect);

        // Height and width  of the glyph in pixels on the raster
        let width = (bbox.width() * self.config.raster_scale).ceil() as u32;
        let height = (bbox.height() * self.config.raster_scale).ceil() as u32;

        let (packed, packed_layer) = self
            .packer
            .pack(glyph_id, width + 2 * padding, height + 2 * padding)
            .ok_or(SMFError::PackingError)?;

        let glyph = GlyphRasterInfo {
            id: glyph_id,
            packed,
            packed_layer,
            padding,
            width,
            height,
            bbox,
        };

        match self.config.raster_kind {
            RasterKind::Bitmap => self.rasterize_bitmap(glyph),
            RasterKind::SDF => self.rasterize_sdf(glyph),
            RasterKind::MSDF => self.rasterize_msdf(glyph),
        }?;

        self.textures.record_texture_update(packed_layer, packed);

        Ok(Some((
            packed.map(|v| v as f32 / self.gconf.atlas_size as f32),
            packed_layer,
        )))
    }

    pub fn name(&self) -> String {
        self.face
            .names()
            .get(0)
            .and_then(|n| n.to_string())
            .unwrap_or_else(|| "Unnamed Font".to_string())
    }
}

struct Builder {
    /// Rasterizer
    rasterizer: Rasterizer,
    /// Point the last segment ended on
    last_point: Point,
    /// Start of the contour
    first_point: Point,
    /// BoundingBox for the points in shape space
    bbox: BoundingBox<f32>,
    /// Padding added on both sides (in pixels)
    padding: f32,
    /// Unpadded width of the glyph's raster (in pixels)
    width: f32,
    /// Unpadded height of the glyph's raster (in pixels)
    height: f32,
}

impl Builder {
    fn new(glyph_padding: u32, bbox: BoundingBox<f32>, width: u32, height: u32) -> Self {
        Self {
            first_point: Point { x: 0.0, y: 0.0 },
            last_point: Point { x: 0.0, y: 0.0 },
            bbox,
            rasterizer: Rasterizer::new(
                (width + 2 * glyph_padding) as usize,
                (height + 2 * glyph_padding) as usize,
            ),
            padding: glyph_padding as f32,
            width: width as f32,
            height: height as f32,
        }
    }
    fn point(&self, x: f32, y: f32) -> Point {
        Point {
            x: (x - self.bbox.x1) * self.width / self.bbox.width() + self.padding,
            y: (y - self.bbox.y1) * self.height / self.bbox.height() + self.padding,
        }
    }
}

impl OutlineBuilder for Builder {
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p1 = self.point(x1, y1);
        let p2 = self.point(x2, y2);
        let p3 = self.point(x, y);
        self.rasterizer.draw_cubic(self.last_point, p1, p2, p3);
        self.last_point = p3;
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p1 = self.point(x1, y1);
        let p2 = self.point(x, y);
        self.rasterizer.draw_quad(self.last_point, p1, p2);
        self.last_point = p2;
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p1 = self.point(x, y);
        self.rasterizer.draw_line(self.last_point, p1);
        self.last_point = p1;
    }
    fn move_to(&mut self, x: f32, y: f32) {
        let p1 = self.point(x, y);
        self.first_point = p1;
        self.last_point = p1;
    }
    fn close(&mut self) {
        if self.last_point != self.first_point {
            self.rasterizer.draw_line(self.last_point, self.last_point);
        }
    }
}
