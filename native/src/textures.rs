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

use std::{any::Any, path::Path, pin::Pin};

use image::{GenericImage, ImageBuffer, Luma, Pixel, Rgb, GenericImageView, EncodableLayout, PixelWithColorType};
use num_traits::Zero;

use crate::{
    config::GlobalConfig,
    font::{Font, RasterKind},
    packing::BoundingBox,
};

pub enum Diff {
    textureUpdate {
        tex: u32,
        channel_count: u8,
        damage: BoundingBox<u32>,
        size: u32,
        data: *const (),
    },
    textureCreation {
        tex: u32,
        channel_count: u8,
        size: u32,
        data: *const (),
    },
}

impl Diff {
    fn tex(&self) -> u32 {
        match self {
            Diff::textureCreation { tex, .. } => *tex,
            Diff::textureUpdate { tex, .. } => *tex,
        }
    }
}

/// The textures data associated with a font struct
pub struct TextureStore<P: Pixel> {
    size: u32,
    textures: Vec<ImageBuffer<P, Pin<Box<[P::Subpixel]>>>>,
    diffs: Vec<Diff>,
}

impl<P: Pixel> TextureStore<P> {
    pub fn new(size: u32) -> Self {
        Self {
            size,
            textures: Vec::new(),
            diffs: Vec::new(),
        }
    }

    fn create_texture(&mut self) {
        let size = self.size;
        let index = self.textures.len();
        let container = Box::into_pin(
            vec![
                <<P as Pixel>::Subpixel as Zero>::zero();
                size as usize * size as usize * P::CHANNEL_COUNT as usize
            ]
            .into_boxed_slice(),
        );
        let data = container.as_ptr() as *const ();
        self.textures
            .push(ImageBuffer::from_raw(size, size, container).unwrap());
        self.diffs.push(Diff::textureCreation {
            tex: index as u32,
            channel_count: P::CHANNEL_COUNT,
            size,
            data,
        });
    }

    /// Get the texture at an index, will create it and all previous ones if doesn't yet exist
    fn get_texture(&mut self, index: u32) -> &mut ImageBuffer<P, Pin<Box<[P::Subpixel]>>> {
        let index = index as usize;
        if self.textures.len() <= index {
            for _ in self.textures.len()..=index {
                self.create_texture();
            }
        }

        &mut self.textures[index]
    }
}

impl<P: Pixel + PixelWithColorType + 'static> TextureStoreCommon for TextureStore<P> where [<P as Pixel>::Subpixel]: EncodableLayout {
    fn len(&self) -> usize {
        self.textures.len()
    }

    fn record_texture_update(&mut self, tex: u32, bbox: BoundingBox<u32>) {
        self.diffs.push(Diff::textureUpdate {
            tex,
            channel_count: P::CHANNEL_COUNT,
            damage: bbox,
            size: self.size,
            data: self.textures[tex as usize].as_ptr() as *const (),
        });
    }

    fn write_tex(&mut self, index: u32, path: &Path) {
        let _ = self.get_texture(index).save_with_format(path, image::ImageFormat::Png);
    }

    /// Concat all the diffs that happened since the last take and return them
    fn take_concatenated_diffs(&mut self) -> Vec<Diff> {
        let mut res = Vec::<Diff>::with_capacity(self.len());
        
        for diff in self.diffs.drain(..) {
            if let Some(prev_diff) = res.iter_mut().find(|d| d.tex() == diff.tex()) {
                // This is a diff affecting an already touched texture: update it
                match diff {
                    Diff::textureCreation { .. }  => {
                        // Technically an unreachable case, but this is the way it should be
                        // handled if it was possible
                        let _ = std::mem::replace(prev_diff, diff);
                    },
                    Diff::textureUpdate { damage: new_damage, .. } => match prev_diff {
                        Diff::textureCreation { .. } => {  
                            // Do nothing, the texture will be loaded entirely anyways so whether
                            // is has been updated and where doesn't matter
                        },
                        Diff::textureUpdate { damage, .. } => {
                            // We have two texture updates, merge them
                            damage.wrap_box(new_damage);
                        }
                    }
                }
            } else {
                // This is a diff affecting an untouched texture, just add it as is
                res.push(diff);
            }
        }

        res
    }
}

impl GenericTextureStore for TextureStore<Rgb<u8>> {
    fn get_texture_rgb(&mut self, index: u32) -> Option<&mut ImageBuffer<Rgb<u8>, Pin<Box<[u8]>>>> {
        Some(self.get_texture(index))
    }

    fn get_texture_luma(
        &mut self,
        _index: u32,
    ) -> Option<&mut ImageBuffer<Luma<u8>, Pin<Box<[u8]>>>> {
        None
    }
}

impl GenericTextureStore for TextureStore<Luma<u8>> {
    fn get_texture_luma(
        &mut self,
        index: u32,
    ) -> Option<&mut ImageBuffer<Luma<u8>, Pin<Box<[u8]>>>> {
        Some(self.get_texture(index))
    }

    fn get_texture_rgb(
        &mut self,
        _index: u32,
    ) -> Option<&mut ImageBuffer<Rgb<u8>, Pin<Box<[u8]>>>> {
        None
    }
}

pub trait TextureStoreCommon: 'static {
    fn write_tex(&mut self, index: u32, path: &Path);
    fn len(&self) -> usize;
    fn record_texture_update(&mut self, tex: u32, bbox: BoundingBox<u32>);
    fn take_concatenated_diffs(&mut self) -> Vec<Diff>;
}

pub trait GenericTextureStore: 'static + TextureStoreCommon {
    fn get_texture_rgb(&mut self, index: u32) -> Option<&mut ImageBuffer<Rgb<u8>, Pin<Box<[u8]>>>>;
    fn get_texture_luma(
        &mut self,
        index: u32,
    ) -> Option<&mut ImageBuffer<Luma<u8>, Pin<Box<[u8]>>>>;
}
