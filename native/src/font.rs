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
    ops::{Deref, DerefMut},
};

use ab_glyph_rasterizer::{Point, Rasterizer};
use fdsm::{
    generate::{generate_msdf, generate_sdf},
    render::{correct_sign_msdf, correct_sign_sdf},
    shape::Shape,
    transform::Transform,
};
use image::{GenericImage, GenericImageView, ImageBuffer, Luma, Pixel, Primitive, Rgb, SubImage};
use nalgebra::{Affine2, Scale2, Similarity2, TAffine, Transform2, Translation2, Vector2};
use rustybuzz::{Direction, Face as BuzzFace, Language, Script, ShapePlan, UnicodeBuffer};
use ttf_parser::{fonts_in_collection, Face, FaceParsingError, GlyphId, OutlineBuilder};

use crate::{
    config::{FontConfig, GlobalConfig},
    error::SMFError,
    packing::{BoundingBox, Heuristics, Layered, LayeredOnlinePacker, MaxRectPacker},
    textures::{GenericTextureStore, TextureStore},
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
    language: Language,
    script: Script,
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

    fn shape(&mut self, str: &str, cfg: &ShapePlanConfig) {
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
        buffer.set_direction(cfg.direction);
        buffer.set_script(cfg.script);

        let buffer = rustybuzz::shape_with_plan(&self.face, plan, buffer);
    }

    // TODO: remove pub

    /// Get a glyph's location in the atlasses, rasterizing it if it isn't already.
    pub fn get_glyph_location(
        &mut self,
        glyph_id: u32,
    ) -> Result<(BoundingBox<f32>, u32), SMFError> {
        if let Some((bbox, layer)) = self.packer.get_loc(&glyph_id) {
            Ok((bbox.map(|v| v as f32 / self.gconf.atlas_size as f32), layer))
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
    fn rasterize_glyph(&mut self, raw_glyph_id: u32) -> Result<(BoundingBox<f32>, u32), SMFError> {
        let glyph_id = GlyphId(raw_glyph_id as u16);
        let Some(bbox) = self.face.glyph_bounding_box(glyph_id) else {
            return if let Some(_) = self.face.glyph_svg_image(glyph_id) {
                // TODO: Handle SVG glyphs
                Err(SMFError::UnsupportedGlyphFormat)
            } else if let Some(_) = self.face.glyph_raster_image(glyph_id, 0) {
                // TODO: Handle raster glyphs
                Err(SMFError::UnsupportedGlyphFormat)
            } else {
                // NOTE: This is probably wrong but we'll catch that later
                // -> We need a way to handle missing glyphs with font fallbacks and what not
                Err(SMFError::ExtraError(
                    "Glyph has no outline, no raster and no SVG".to_string(),
                ))
            };
        };

        let padding = self.gconf.glyph_padding;

        // Glyph bounding box, in shape space (what we'll call the outline's raw coordinate space)
        let bbox = BoundingBox::new(
            bbox.x_min as f32,
            bbox.y_min as f32,
            bbox.x_max as f32,
            bbox.y_max as f32,
        );

        // Height and width  of the glyph in pixels on the raster
        let width = (bbox.width() * self.config.scale).ceil() as u32;
        let height = (bbox.height() * self.config.scale).ceil() as u32;

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

        Ok((
            packed.map(|v| v as f32 / self.gconf.atlas_size as f32),
            packed_layer,
        ))
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

#[cfg(test)]
mod Test {
    use std::path::Path;

    use crate::config::GlobalConfig;

    use super::Font;

    #[test]
    fn show_texs() {
        println!("begin");
        let mut cfg = GlobalConfig::get_copy();

        cfg.atlas_size = 4096;
        cfg.sin_alpha = 0.03;
        cfg.glyph_padding = 8;
        cfg.coloring_seed = 1234345345654364356;

        cfg.update();

        println!("set gconf");

        let data = &std::fs::read("../src/main/resources/assets/fonts/font.ttc").unwrap();
        let scale = 1.0 / 8.0;

        let fonts = [
            Font::from_bytes_and_index(
                data,
                0,
                crate::config::FontConfig {
                    raster_kind: super::RasterKind::Bitmap,
                    scale,
                },
            )
            .unwrap(),
            Font::from_bytes_and_index(
                data,
                0,
                crate::config::FontConfig {
                    raster_kind: super::RasterKind::SDF,
                    scale,
                },
            )
            .unwrap(),
            Font::from_bytes_and_index(
                data,
                0,
                crate::config::FontConfig {
                    raster_kind: super::RasterKind::MSDF,
                    scale,
                },
            )
            .unwrap(),
        ];

        println!("loaded fonts");

        let str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        for (i, mut font) in fonts.into_iter().enumerate() {
            for c in str.chars() {
                let id = font.face.glyph_index(c).unwrap_or(ttf_parser::GlyphId(1));
                println!("font {i} char {c} id {id:?}");
                font.get_glyph_location(id.0.into()).unwrap();
            }

            println!("added chars to font {i}");

            let path_str = format!("./tex{i}.png");
            let path = Path::new(&path_str);
            font.textures.write_tex(0, path);

            println!("wrote font {i}");
        }
    }
}
