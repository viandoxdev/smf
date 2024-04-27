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

use std::{
    collections::HashMap,
    ffi::c_char,
    fmt::Display,
    io::Write,
    iter,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
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

#[derive(Default)]
struct LineBreakInfo {
    inner: Vec<(usize, BreakOpportunity)>,
}

impl LineBreakInfo {
    fn new(str: &str) -> Self {
        let inner = unicode_linebreak::linebreaks(str).collect::<Vec<_>>();
        Self { inner }
    }

    #[inline(always)]
    fn get_index(&self, index: usize) -> Result<usize, usize> {
        self.inner.binary_search_by_key(&index, |(i, _)| *i)
    }

    /// Get the opportunity at (right before) the index
    fn opportunity(&self, index: usize) -> Option<BreakOpportunity> {
        Some(self.inner[self.get_index(index).ok()?].1)
    }

    /// Get the closest opportunity before an index
    fn last_opportunity(&self, index: usize) -> Option<(usize, BreakOpportunity)> {
        match self.get_index(index) {
            // Theres an opportunity at this exact index (read right before that index), so return
            // that
            Ok(index) => Some(self.inner[index]),
            // There is no opportunity at this exact index, so get_index returns the index of the
            // opportunity right after that, which means the closest one before the index is the
            // one before (or none if there isn't any)
            Err(index) => Some(self.inner[index.checked_sub(1)?]),
        }
    }

    /// Set the / add an opportunity at (just before) an index
    fn set_opportunity(&mut self, index: usize, op: BreakOpportunity) {
        match self.get_index(index) {
            Ok(idx) => {
                self.inner[idx] = (index, op);
            }
            Err(idx) => {
                self.inner.insert(idx, (index, op));
            }
        }
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

struct PreLayoutGlyph {
    info: GlyphInfo,
    pos: GlyphPosition,
    /// Whether this glyph is the last of its line
    end_of_line: bool,
    level: Level,
    base_level: Level,
}

struct ShapedRun<'a> {
    buffer: GlyphBuffer,
    str: &'a str,
    start: usize,
    end: usize,
    /// Whether this run is the last of its line (mandatory break only)
    end_of_line: bool,
    config: ShapePlanConfig,
    level: Level,
    base_level: Level,
}

impl<'a> ShapedRun<'a> {
    fn new(
        font: &mut Font,
        str: &'a str,
        start: usize,
        end: usize,
        config: ShapePlanConfig,
        end_of_line: bool,
        level: Level,
        base_level: Level,
    ) -> Self {
        let str = &str[start..end];
        let buffer = font.shape(str, &config);
        Self {
            buffer,
            str,
            start,
            end,
            end_of_line,
            config,
            base_level,
            level,
        }
    }
}

pub struct Font<'a, P: LayeredOnlinePacker<u32> = Layered<MaxRectPacker<u32>, u32>> {
    shape_plan_chache: HashMap<ShapePlanConfig, ShapePlan>,
    face: BuzzFace<'a>,
    packer: P,
    pub config: FontConfig,
    gconf: GlobalConfig,
    textures: Box<dyn GenericTextureStore>,
}

impl<'f> Font<'f> {
    pub fn from_bytes<'a: 'f>(bytes: &'a [u8], config: FontConfig) -> Result<Vec<Self>, SMFError> {
        let n = fonts_in_collection(bytes).ok_or(SMFError::FailedParsing(None))?;

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
        let (runs, _) = self.split_runs(str, false);
        let mut lines = self.pre_layout(&runs);
        self.layout(&mut lines)
    }

    pub fn process_multiline(
        &mut self,
        str: &str,
        max_length: Option<f32>,
    ) -> Result<Vec<Mesh>, SMFError> {
        println!("<multiline> '{str}' ({max_length:?})");
        println!("<multiline> legend:");
        println!("<multiline>   base direction: \x1b[44m \x1b[0m LTR \x1b[45m \x1b[0m RTL");
        println!("<multiline>   direction:      \x1b[31ma\x1b[0m LTR \x1b[32ma\x1b[0m RTL");
        println!("<multiline>   linebreaks:     \x1b[31ma\x1b[0m unallowed \x1b[32ma\x1b[0m allowed \x1b[36ma\x1b[0m mandatory");
        let (mut runs, linebreaks) = self.split_runs(str, true);
        {
            let mut res = String::new();
            let mut last_ended = false;
            for (i, run) in runs.iter().enumerate() {
                if i > 0 && !last_ended {
                    res.push_str("\x1b[0;37m|");
                }
                if run.base_level.is_ltr() {
                    res.push_str("\x1b[0;44m");
                } else {
                    res.push_str("\x1b[0;45m");
                }
                if run.level.is_ltr() {
                    res.push_str("\x1b[31m");
                } else {
                    res.push_str("\x1b[32m");
                }
                res.push_str(run.str);
                if run.end_of_line {
                    res.push_str("\x1b[0;37m\\n");
                }
                last_ended = run.end_of_line;
            }

            println!("<multiline> runs: '\u{202d}{res}\x1b[0m'");
        }
        {
            let mut res = String::new();
            for (i, c) in str.char_indices() {
                match linebreaks.opportunity(i) {
                    Some(BreakOpportunity::Allowed) => res.push_str("\x1b[32m|\x1b[31m"),
                    Some(BreakOpportunity::Mandatory) => res.push_str("\x1b[36m|\x1b[31m"),
                    _ => {},
                }
                res.push(c);
            }
            println!("<multiline> linebreaks: '\u{202d}\x1b[31m{res}\x1b[0m'");
        }
        let max_length = max_length.unwrap_or(f32::MAX);
        let mut lines = self.pre_layout_multiline(str, &mut runs, linebreaks, max_length);
        {
            let mut res = String::new();
            for glyph in &lines {
                if glyph.base_level.is_ltr() {
                    res.push_str("\x1b[0;44m");
                } else {
                    res.push_str("\x1b[0;45m");
                }
                if glyph.level.is_ltr() {
                    res.push_str("\x1b[31m");
                } else {
                    res.push_str("\x1b[32m");
                }
                let c = glyph.info.cluster as usize;
                res.push_str(&str[c..=c]);
                if glyph.end_of_line {
                    res.push_str("\x1b[0;37m\\n");
                }
            }

            println!("<multiline> lines: '\u{202d}{res}\x1b[0m'");
        }
        self.layout(&mut lines)
    }

    // TODO: Figure language out through configurable language list and script value rather than
    // just requesting it (see: https://github.com/harfbuzz/harfbuzz/issues/1288)

    /// Split text into runs (belonging to the same paragraph and script), and shape them.
    /// This is done before layout so depending on what happens after some text might be reshaped.
    fn split_runs<'a: 's, 's>(
        &'s mut self,
        str: &'a str,
        multiline: bool,
    ) -> (Vec<ShapedRun<'a>>, LineBreakInfo) {
        let bidi = BidiInfo::new(str, None);
        let linebreak = if multiline {
            LineBreakInfo::new(str)
        } else {
            LineBreakInfo::default()
        };

        let mut runs = Vec::new();

        // Split text into runs of similar Scripts (and paragraphs) and shape them.
        for par in &bidi.paragraphs {
            let par_start = par.range.start;
            let par_str = &str[par.range.clone()];

            let mut level = par.level;
            let mut start = 0usize;
            let mut scripts = ScriptExtension::default();
            // Iterator over each character's position and its corresponding scripts, with an extra
            // Unknown (empty set) script added to end the last run.
            let scripts_iter = par_str
                .char_indices()
                .map(|(i, c)| (i, c.script_extension()))
                .chain(std::iter::once((par_str.len(), Script::Unknown.into())));
            for (i, char_scripts) in scripts_iter {
                let next_scripts = scripts.intersection(char_scripts);
                // Whether a line break should occur before the current character
                // We want to make sure runs are split by linebreaks as well to avoid shaping
                // accros lines (we would need to reshape in layout otherwise, which is slower)
                let should_break = (multiline
                    && i > 0
                    && matches!(linebreak.opportunity(i), Some(BreakOpportunity::Mandatory)))
                    || (i >= par_str.len());

                let next_level = if i < par_str.len() {
                    bidi.levels[i]
                } else {
                    level
                };
                let level_changed = next_level != level;

                // Check if the current character isn't compatible with the run or if we need to
                // break before the current character
                if next_scripts.is_empty() || should_break || level_changed {
                    // If so, the current run ends with the previous char (since the current one
                    // is incompatible), so process that run
                    {
                        let str = &par_str[start..i];
                        let script = scripts
                            .iter()
                            .next()
                            .expect("ScriptExtension isn't empty, yet iterator yields no script");
                        let script = rustybuzz::Script::from_str(script.short_name())
                            .expect("Couldn't convert script to rustybuzz");
                        let language = self.config.language.clone();
                        let cfg = ShapePlanConfig {
                            direction: level_to_dir(level),
                            language,
                            script,
                        };

                        let buffer = self.shape(str, &cfg);

                        runs.push(ShapedRun {
                            buffer,
                            str,
                            start: par_start + start,
                            end: par_start + i,
                            end_of_line: should_break,
                            config: cfg,
                            base_level: par.level,
                            level,
                        });
                    }
                    // Start a new run
                    start = i;
                }

                scripts = if next_scripts.is_empty() {
                    char_scripts
                } else {
                    next_scripts
                };
                level = next_level;
            }
        }

        (runs, linebreak)
    }

    /// Just turn runs into PreLayoutGlyph
    fn pre_layout<'a>(&mut self, runs: &[ShapedRun<'a>]) -> Vec<PreLayoutGlyph> {
        let mut res = Vec::new();

        for run in runs {
            for (&info, &pos) in run
                .buffer
                .glyph_infos()
                .iter()
                .zip(run.buffer.glyph_positions().iter())
            {
                res.push(PreLayoutGlyph {
                    pos,
                    info,
                    level: run.level,
                    base_level: run.base_level,
                    end_of_line: false,
                });
            }
        }

        res.last_mut().unwrap().end_of_line = true;

        res
    }

    /// Perform text wrap and reshape as needed
    fn pre_layout_multiline<'a>(
        &mut self,
        str: &'a str,
        runs: &'a mut Vec<ShapedRun<'a>>,
        mut lb: LineBreakInfo,
        max_length: f32,
    ) -> Vec<PreLayoutGlyph> {
        let mut x = 0f32;

        // NOTE: (Bikeshed) this is a shit name, but the idea is that GlyphCheckpoints back up the
        // state of the "line wrapping machine" so that if we decide the break the line at an
        // index, we can restart the machine back at the break point, kind of like a checkpoint in
        // a video game
        #[derive(Debug, Clone, Copy)]
        struct GlyphCheckpoint {
            /// Index of the start of the glyph's cluster in the string
            index: usize,
            /// Index of the run in runs
            run_index: usize,
            /// Index of the glyph in the run's glyph_infos
            glyph_index: usize,
            /// x offset before the glyph was placed
            x: f32,
            level: Level,
            base_level: Level,
        }

        let mut current_line = Vec::<GlyphCheckpoint>::new();

        let mut lines = Vec::new();

        // We can't use a for loop since we'll be "going back in time" if necessary
        let mut run_index = 0usize;
        let mut glyph_index = 0usize;
        'run_loop: while run_index < runs.len() {
            let run = &runs[run_index];

            while glyph_index < run.buffer.glyph_infos().len() {
                let pos = run.buffer.glyph_positions()[glyph_index];
                let glyph = run.buffer.glyph_infos()[glyph_index];

                let c = glyph.cluster as usize + run.start;
                println!("Placing '{}' ({c})", str[c..].chars().next().unwrap());

                let adv = pos.x_advance as f32 * self.scale();

                let is_first_glyph_of_line = current_line.is_empty();

                current_line.push(GlyphCheckpoint {
                    index: glyph.cluster as usize + run.start,
                    run_index,
                    glyph_index,
                    x,
                    level: run.level,
                    base_level: run.base_level,
                });

                // Breaking on the first glyph of the line doesn't make sense because it would just
                // end up the first glyph of the next line (and overflow there too) causing an
                // infinite loop
                if x + adv > max_length && !is_first_glyph_of_line {
                    // We exceeded the line's max length, we need to break (and potentially
                    // reshape some runs)

                    // Figure out where we need to break
                    // We should break right before break_index
                    let (break_index, safe) =
                        match lb.last_opportunity(glyph.cluster as usize + run.start) {
                            Some((index, BreakOpportunity::Allowed)) => {
                                // We can, and should, break here.
                                // Check if it is safe to break here: try to look for the glyph whose
                                // cluster starts at index, and check if the corresponding glyph can be
                                // broken, if not found, just assume it isn't safe
                                let safe = runs[current_line[0].run_index..=run_index]
                                    .iter()
                                    .flat_map(|r| {
                                        r.buffer
                                            .glyph_infos()
                                            .iter()
                                            .zip(std::iter::repeat(r.start))
                                    })
                                    .any(|(g, start)| {
                                        g.cluster as usize + start == index && !g.unsafe_to_break()
                                    });
                                (index, safe)
                            }
                            _ => {
                                // Either the last opportunity was a mandatory one (which mean it
                                // wasn't in this line), or there was no opportunity before. This means
                                // we need to force break somewhere.

                                // Just break before this cluster
                                (glyph.cluster as usize + run.start, !glyph.unsafe_to_break())
                            }
                        };

                    // Record a mandatory line break to make sure further iterations are aware of
                    // it
                    lb.set_opportunity(break_index, BreakOpportunity::Mandatory);

                    if safe {
                        // Because we conservatively assume that if a break has to happen mid
                        // cluster, it isn't safe, we only have to handle the case where
                        // break_index aligns with a cluster here
                        
                        println!("looking for {break_index}");
                        println!("'{:?}'", str.char_indices().collect_vec());
                        println!("{:?}", current_line.iter().map(|g| g.index).collect_vec());

                        // break_glyph_index is the index of the glyph before which we break in
                        // current_line, break_glyph is the corresponding GlyphCheckpoint

                        // This finds the glyph in the line whose index is closest (but <=) than
                        // break_index. The reason is that because of how bidi text works, we might
                        // try to break on a glyph that isn't yet in current_line (one that is
                        // logically before the current one, but is after in display order), as
                        // such we can't find that glyph in current line. Example:
                        // str: "hello WORLD" (logical order) (uppercases represent RTL text)
                        // runs: "hello ", "WORLD"
                        // Say we have just enough space for "hello DL", we would get:
                        // current_line: "hello DL", but we would want to break on "W"
                        // which hasn't been placed yet. Still "hello " should be put on a separate
                        // line, so we need to 
                        let (break_glyph_index, &break_glyph, _) = current_line
                            .iter()
                            .enumerate()
                            .map(|(i, glyph)| (i, glyph, break_index.checked_sub(glyph.index).unwrap_or(usize::MAX)))
                            .min_by_key(|(_, _, diff)| *diff)
                            .expect("Can't find break cluster in line");

                        // Perform the break
                        lines.extend(current_line[0..break_glyph_index].iter().map(|g| {
                            let buf = &runs[g.run_index].buffer;
                            PreLayoutGlyph {
                                pos: buf.glyph_positions()[g.glyph_index],
                                info: buf.glyph_infos()[g.glyph_index],
                                end_of_line: false,
                                level: g.level,
                                base_level: g.base_level,
                            }
                        }));
                        lines.last_mut().unwrap().end_of_line = true;

                        // We now need to reproccess any glyph from the break point to where we are
                        // now, so we reset the state to what it was at the break point
                        x = 0.0;
                        current_line.clear();
                        run_index = break_glyph.run_index;
                        glyph_index = break_glyph.glyph_index;

                        // This is pretty much a goto, I really need a different design for this
                        continue 'run_loop;
                    } else {
                        // We can't safely break, so we need to get the broken run, split it, and
                        // reshape each part
                        // Here break index may not exactly align with any cluster (I mean I'm not
                        // actually sure so I just assume it doesn't have to)

                        // Look for the run where the break happens
                        // Start looking from the first run of the line
                        let (broken_run_index, broken_run) = runs[current_line[0].run_index..]
                            .iter()
                            .find_position(|r| break_index >= r.start && break_index < r.end)
                            .expect("Can't find break run");

                        // Reshape runs
                        let run_before_break = ShapedRun::new(
                            self,
                            str,
                            broken_run.start,
                            break_index,
                            broken_run.config.clone(),
                            true,
                            broken_run.level,
                            broken_run.base_level,
                        );
                        let run_after_break = ShapedRun::new(
                            self,
                            str,
                            break_index,
                            broken_run.end,
                            broken_run.config.clone(),
                            false,
                            broken_run.level,
                            broken_run.base_level,
                        );

                        // Replace broken run with both runs
                        runs.insert(broken_run_index, run_before_break);
                        runs[broken_run_index + 1] = run_after_break;

                        let line_start_of_run = current_line
                            .iter()
                            .position(|g| g.run_index == broken_run_index)
                            .expect("Break run can't be found in current_line");

                        run_index = broken_run_index;
                        glyph_index = 0;
                        x = current_line[line_start_of_run].x;

                        // Remove all of the elements of the broken run and after since their
                        // layout is bad
                        current_line.drain(line_start_of_run..);

                        continue 'run_loop;
                    }
                } else {
                    x += adv;
                }

                glyph_index += 1;
            }

            glyph_index = 0;

            if run.end_of_line {
                lines.extend(current_line.drain(..).map(|g| {
                    let buf = &runs[g.run_index].buffer;
                    PreLayoutGlyph {
                        pos: buf.glyph_positions()[g.glyph_index],
                        info: buf.glyph_infos()[g.glyph_index],
                        end_of_line: false,
                        level: g.level,
                        base_level: g.base_level,
                    }
                }));
                lines.last_mut().unwrap().end_of_line = true;
                x = 0.0;
            }

            run_index += 1;
        }

        lines
    }

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
        let mut base_level = lines
            .first()
            .map(|g| g.base_level)
            .unwrap_or(Level::ltr());

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
                let loc = self.get_glyph_location(g.info.glyph_id)?;
                let bbox = self
                    .face
                    .glyph_bounding_box(GlyphId(g.info.glyph_id as u16))
                    .map(rect_to_bbox)
                    .unwrap_or(BoundingBox::ZERO);

                // TODO: Is this actually doing anything ?
                let kerning = if let (Some(last_glyph), Some(subtables)) = (last_glyph, subtables) {
                    let kern = subtables
                        .into_iter()
                        .flat_map(|t| t.glyphs_kerning(last_glyph, GlyphId(g.info.glyph_id as u16)))
                        .next()
                        .unwrap_or_default();
                    kern as f32 * self.scale()
                } else {
                    0.0
                };

                last_glyph = Some(GlyphId(g.info.glyph_id as u16));

                let (gx, gy) = (
                    x + g.pos.x_offset as f32 * self.scale() + kerning,
                    y + g.pos.y_offset as f32 * self.scale(),
                );

                x += g.pos.x_advance as f32 * self.scale();

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

        rustybuzz::shape_with_plan(&self.face, plan, buffer)
    }

    /// Get a glyph's location in the atlasses, rasterizing it if it isn't already.
    fn get_glyph_location(
        &mut self,
        glyph_id: u32,
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
        raw_glyph_id: u32,
    ) -> Result<Option<(BoundingBox<f32>, u32)>, SMFError> {
        let glyph_id = GlyphId(raw_glyph_id as u16);
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
            .pack(raw_glyph_id, width + 2 * padding, height + 2 * padding)
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
