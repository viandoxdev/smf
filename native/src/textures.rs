// Notes:
// A font contains a set of glyphs that need to be rasterized
//  -> Different font format (-> != memory layouts) depending on raster kind
//  -> At least one special texture for raster glyphs (emojis)
//    -> potentially separate packer for these ?
//
// Or we use the same format for all textures and figure everything out in the shader
//   -> Risk of top level branching which isn't good IIRC (All fragments should try to follow
//   roughly the same code path)
//
// Also this all probably means separate mesh / draw calls per cluster of characters that are on
// the same texture (or find a way to bypass that issue, texture arrays maybe, but sounds like a
// pain)
//
// => For now: we ignore raster glyphs (too much work for a MVP, ill see about them later)
//   -> different draw calls for each texture within a single text layout

// TODO: Mipmaps

// TODO: Raster glyphs

use std::{any::Any, path::Path};

use image::{GenericImage, ImageBuffer, Pixel, Rgb, Luma};

use crate::{config::GlobalConfig, font::{RasterKind, Font}, packing::BoundingBox};

//#[derive(Debug, Clone, Copy, PartialEq, Eq)]
//pub enum TextureFormat {
//    Rgba32,
//    R8,
//}
//
//#[derive(Debug, Clone, Copy, PartialEq, Eq)]
//pub enum PixelValue {
//    Rgba32(u8, u8, u8, u8),
//    R8(u8),
//}
//
//impl TextureFormat {
//    /// Size of a pixel in bytes
//    pub fn pixel_size(&self) -> u32 {
//        match self {
//            Self::R8 => 1,
//            Self::Rgba32 => 4,
//        }
//    }
//}
//
//impl From<RasterKind> for TextureFormat {
//    fn from(value: RasterKind) -> Self {
//        match value {
//            RasterKind::Bitmap => Self::R8,
//            RasterKind::SDF => Self::R8,
//            RasterKind::MSDF => Self::Rgba32,
//        }
//    }
//}

//pub struct Texture {
//    pub size: u32,
//    pub format: TextureFormat,
//    buffer: ImageBuffer<>
//}

//impl Texture {
//    /// Just size as usize
//    pub fn capacity(&self) -> usize {
//        self.size as usize
//    }
//
//    /// Pointer to the start of texture's data
//    pub unsafe fn ptr(&self) -> *const u8 {
//        self.bytes.as_ptr()
//    }
//
//    pub fn set_pixel(&mut self, x: u32, y: u32, value: PixelValue) -> Option<()> {
//        if x >= self.size || y >= self.size {
//            return None;
//        }
//
//        let off = (x + y * self.size * self.format.pixel_size()) as usize;
//
//        match self.format {
//            TextureFormat::R8 => {
//                let PixelValue::R8(v) = value else {
//                    return None;
//                };
//
//                self.bytes[off] = v;
//
//                Some(())
//            }
//            TextureFormat::Rgba32 => {
//                let PixelValue::Rgba32(r, g, b, a) = value else {
//                    return None;
//                };
//
//                self.bytes[off] = r;
//                self.bytes[off + 1] = g;
//                self.bytes[off + 2] = b;
//                self.bytes[off + 3] = a;
//
//                Some(())
//            }
//        }
//    }
//
//    pub fn get_mut_view(&mut self, bbox: BoundingBox<u32>) -> TextureView {
//
//    }
//}

/// The textures data associated with a font struct
pub struct TextureStore<P: Pixel> {
    size: u32,
    textures: Vec<ImageBuffer<P, Vec<P::Subpixel>>>
}

impl<P: Pixel> TextureStore<P> {
    pub fn new(size: u32) -> Self {
        Self {
            size,
            textures: Vec::new(),
        }
    }

    fn create_texture(&mut self) {
        let size = self.size;
        self.textures.push(ImageBuffer::new(size, size));
    }

    /// Get the texture at an index, will create it and all previous ones if doesn't yet exist
    pub fn get_texture(&mut self, index: u32) -> &mut ImageBuffer<P, Vec<P::Subpixel>> {
        let index = index as usize;
        if self.textures.len() <= index {
            for _ in self.textures.len()..=index {
                self.create_texture();
            }
        }

        &mut self.textures[index]
    }
}

impl GenericTextureStore for TextureStore<Rgb<u8>> {
    fn get_texture_rgb(&mut self, index: u32) -> Option<&mut ImageBuffer<Rgb<u8>, Vec<u8>>> {
        Some(self.get_texture(index))
    }

    fn get_texture_luma(&mut self, _index: u32) -> Option<&mut ImageBuffer<Luma<u8>, Vec<u8>>> {
        None
    }

    fn write_tex(&mut self, index: u32, path: &Path) {
        let _ = self.get_texture(index).save_with_format(path, image::ImageFormat::Png);
    }
}

impl GenericTextureStore for TextureStore<Luma<u8>> {
    fn get_texture_luma(&mut self, index: u32) -> Option<&mut ImageBuffer<Luma<u8>, Vec<u8>>> {
        Some(self.get_texture(index))
    }

    fn get_texture_rgb(&mut self, _index: u32) -> Option<&mut ImageBuffer<Rgb<u8>, Vec<u8>>> {
        None
    }

    fn write_tex(&mut self, index: u32, path: &Path) {
        let _ = self.get_texture(index).save_with_format(path, image::ImageFormat::Png);
    }
}


pub trait GenericTextureStore: 'static {
    fn get_texture_rgb(&mut self, index: u32) -> Option<&mut ImageBuffer<Rgb<u8>, Vec<u8>>>;
    fn get_texture_luma(&mut self, index: u32) -> Option<&mut ImageBuffer<Luma<u8>, Vec<u8>>>;
    fn write_tex(&mut self, index: u32, path: &Path);
}
