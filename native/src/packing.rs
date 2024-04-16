/// 2d rectangle packing utilities
/// Used in the context of atlas generation
use std::{
    collections::HashMap,
    hash::Hash,
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub}, fmt::Display,
};

use self::zero_const::NumConst;

// TODO:
//  - Rotation support
//  - Guillotine packing
//  - Simplify Packer api to optimize Layered Packers

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBox<T = u32> {
    pub x1: T,
    pub y1: T,
    pub x2: T,
    pub y2: T,
}

// Not sure why this isn't part of num traits yet, There has to be a way to have this as a feature
// disabled by default, but whatever
pub mod zero_const {
    pub trait NumConst {
        const ZERO: Self;
        const ONE: Self;
        const TWO: Self;
        const MIN: Self;
        const MAX: Self;
    }

    macro_rules! impl_numconst {
        ($t:ty: $zero:expr, $one:expr, $two: expr, $min: expr, $max: expr) => {
            impl NumConst for $t {
                const ZERO: Self = $zero;
                const ONE: Self = $one;
                const TWO: Self = $two;
                const MIN: Self = $min;
                const MAX: Self = $max;
            }
        };
    }

    impl_numconst!(u8: 0, 1, 2, u8::MIN, u8::MAX);
    impl_numconst!(i8: 0, 1, 2, i8::MIN, i8::MAX);
    impl_numconst!(u16: 0, 1, 2, u16::MIN, u16::MAX);
    impl_numconst!(i16: 0, 1, 2, i16::MIN, i16::MAX);
    impl_numconst!(u32: 0, 1, 2, u32::MIN, u32::MAX);
    impl_numconst!(i32: 0, 1, 2, i32::MIN, i32::MAX);
    impl_numconst!(u64: 0, 1, 2, u64::MIN, u64::MAX);
    impl_numconst!(i64: 0, 1, 2, i64::MIN, i64::MAX);
    impl_numconst!(u128: 0, 1, 2, u128::MIN, u128::MAX);
    impl_numconst!(i128: 0, 1, 2, i128::MIN, i128::MAX);
    impl_numconst!(usize: 0, 1, 2, usize::MIN, usize::MAX);
    impl_numconst!(isize: 0, 1, 2, isize::MIN, isize::MAX);
    impl_numconst!(f32: 0.0, 1.0, 2.0, f32::MIN, f32::MAX);
    impl_numconst!(f64: 0.0, 1.0, 2.0, f64::MIN, f64::MAX);
}

impl<T> FromIterator<(T, T)> for BoundingBox<T> 
where 
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumConst
        + PartialOrd
        + Copy,
{
    fn from_iter<I: IntoIterator<Item = (T, T)>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let Some((fx, fy)) = iter.next() else { return Self::ZERO };
        let mut bbox = Self::new(fx, fy, fx, fy);
        for (x, y) in iter {
            bbox.wrap_point(x, y);
        }
        bbox
    }
}

impl<T> FromIterator<BoundingBox<T>> for BoundingBox<T> 
where 
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumConst
        + PartialOrd
        + Copy,
{
    fn from_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let Some(mut bbox) = iter.next() else { return Self::ZERO };
        for other in iter {
            bbox.wrap_box(other);
        }
        bbox
    }
}

impl<T> BoundingBox<T> {
    pub fn new(x1: T, y1: T, x2: T, y2: T) -> Self {
        Self { x1, y1, x2, y2 }
    }

    pub fn map<K: Copy, F: FnMut(T) -> K>(self, mut f: F) -> BoundingBox<K> {
        BoundingBox::<K>::new(f(self.x1), f(self.y1), f(self.x2), f(self.y2))
    }
}

impl<T> Display for BoundingBox<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumConst
        + PartialOrd
        + Copy
        + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { x1, y1, .. } = self;
        let width = self.width();
        let height = self.height();
        write!(f, "{width}x{height} at ({x1}, {y1})")
    }
}

impl<T> BoundingBox<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumConst
        + PartialOrd
        + Copy,
{
    pub const ZERO: Self = BoundingBox {
        x1: T::ZERO,
        y1: T::ZERO,
        x2: T::ZERO,
        y2: T::ZERO,
    };

    /// "Empty" bounding box from (MAX, MAX) to (MIN, MIN) (yes in that order).
    pub const EMPTY: Self = BoundingBox {
        x1: T::MAX,
        y1: T::MAX,
        x2: T::MIN,
        y2: T::MIN,
    };

    pub fn new_dim(x: T, y: T, width: T, height: T) -> Self {
        Self {
            x1: x,
            y1: y,
            x2: x + width,
            y2: y + height,
        }
    }

    /// Wrap the point in the bounding box, this can be used to compute a bounding of any shape:
    /// begin with a EMPTY BoundingBox and wrap all the points.
    pub fn wrap_point(&mut self, x: T, y: T) {
        if x < self.x1 {
            self.x1 = x;
        }
        if y < self.y1 {
            self.y1 = y;
        }
        if x > self.x2 {
            self.x2 = x;
        }
        if y > self.y2 {
            self.y2 = y;
        }
    }

    /// Wrap the box in the bounding box
    pub fn wrap_box(&mut self, bbox: Self) {
        if bbox.x1 < self.x1 {
            self.x1 = bbox.x1;
        }
        if bbox.y1 < self.y1 {
            self.y1 = bbox.y1;
        }
        if bbox.x2 > self.x2 {
            self.x2 = bbox.x2;
        }
        if bbox.y2 > self.y2 {
            self.y2 = bbox.y2;
        }
    }

    pub fn intersects(&self, other: &Self) -> bool {
        other.x2 > self.x1 && other.x1 <= self.x2 && other.y2 > self.y1 && other.y1 <= self.y2
    }

    pub fn area(&self) -> T {
        self.width() * self.height()
    }

    pub fn with_dim(self, width: T, height: T) -> Self {
        Self::new_dim(self.x1, self.y1, width, height)
    }

    #[inline(always)]
    pub fn width(&self) -> T {
        self.x2 - self.x1
    }
    #[inline(always)]
    pub fn height(&self) -> T {
        self.y2 - self.y1
    }

    // TODO: Consider inlining this (in <MaxRectPacker as SequentialPacker>::pack) ?

    /// Slice self into overlapping boxes covering the space not touched
    pub fn slice(&self, cut: &Self) -> impl Iterator<Item = BoundingBox<T>> {
        let boxes: [Option<BoundingBox<T>>; 4] = [
            (cut.y1 > self.y1).then(|| BoundingBox::new(self.x1, self.y1, self.x2, cut.y1)),
            (cut.x1 > self.x1).then(|| BoundingBox::new(self.x1, self.y1, cut.x1, self.y2)),
            (cut.y2 < self.y2).then(|| BoundingBox::new(self.x1, cut.y2, self.x2, self.y2)),
            (cut.x2 < self.x2).then(|| BoundingBox::new(cut.x2, self.y1, self.x2, self.y2)),
        ];
        boxes.into_iter().flatten()
    }

    /// Test if self fully contains other
    pub fn contains(&self, other: &Self) -> bool {
        other.x1 >= self.x1 && other.x2 <= self.x2 && other.y1 >= self.y1 && other.y2 <= self.y2
    }

    pub fn scale(&self, s: T) -> Self {
        Self::new(self.x1 * s, self.y1 * s, self.x2 * s, self.y2 * s)
    }

    pub fn translate(&self, x: T, y: T) -> Self {
        Self::new(self.x1 + x, self.y1 + y, self.x2 + x, self.y2 + y)
    }

    pub fn unpad(&self, padding: T) -> Self {
        if self.width() <= padding + padding || self.height() <= padding + padding {
            let x = (self.x1 + self.x2) / T::TWO;
            let y = (self.y1 + self.y2) / T::TWO;
            Self::new(x, y, x, y)
        } else {
            Self::new(
                self.x1 + padding,
                self.y1 + padding,
                self.x2 - padding,
                self.y2 - padding,
            )
        }
    }

    pub fn pad(&self, padding: T) -> Self {
        Self::new(
            self.x1 - padding,
            self.y1 - padding,
            self.x2 + padding,
            self.y2 + padding,
        )
    }
}

impl From<BoundingBox<u32>> for BoundingBox<f32> {
    fn from(value: BoundingBox<u32>) -> Self {
        Self {
            x1: value.x1 as f32,
            y1: value.y1 as f32,
            x2: value.x2 as f32,
            y2: value.y2 as f32,
        }
    }
}

/// Fast Packer, but relatively ineficient when given items of different heights
pub struct ShelfPacker<K> {
    /// Width of the packing area
    width: u32,
    /// Height of the packing area
    height: u32,
    /// Used area
    used: u64,
    /// Map of items and their respective bounding boxes
    packed: HashMap<K, BoundingBox>,
    /// The bounding box of the current shelf
    shelf: BoundingBox,
}

impl<K: Hash + Eq> ShelfPacker<K> {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            used: 0,
            packed: HashMap::new(),
            shelf: BoundingBox::new(0, 0, width, 0),
        }
    }
}

impl<K: Hash + Eq> OnlinePacker<K> for ShelfPacker<K> {
    fn pack(&mut self, item: K, width: u32, height: u32) -> Option<BoundingBox> {
        // Fast path for items with zero area
        if (width == 0 || height == 0) && width <= self.width && height <= self.height {
            let rect = BoundingBox::new_dim(0, 0, width, height);
            self.packed.insert(item, rect);
            return Some(rect);
        }

        // If there's enough space on the current shelf
        if width <= self.shelf.width() && self.height - self.shelf.y1 >= height {
            // Place the item there
            let rect = BoundingBox::new_dim(self.shelf.x1, self.shelf.y1, width, height);
            // Update the shelf
            self.shelf.x1 += width;
            self.shelf.y2 = self.shelf.y2.max(rect.y2);

            self.packed.insert(item, rect);
            self.used += rect.area() as u64;
            Some(rect)
        } else if width <= self.width && self.height - self.shelf.y2 >= height {
            // If we can place the item on a new shelf
            let rect = BoundingBox::new_dim(0, self.shelf.y2, width, height);
            // Update the shelf
            self.shelf.x1 = rect.x2;
            self.shelf.y1 = rect.y1;
            self.shelf.y2 = rect.y2;

            self.packed.insert(item, rect);
            self.used += rect.area() as u64;
            Some(rect)
        } else {
            // Not enough space for the item
            None
        }
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn get_loc(&self, item: &K) -> Option<BoundingBox> {
        self.packed.get(item).copied()
    }

    fn usage(&self) -> f32 {
        self.used as f32 / (self.width as f32 * self.height as f32)
    }
}

// From https://github.com/solomon-b/greedypacker, I originally had the idea myself (on paper),
// but used the above repo as a reference implementation.
/// Packer keeping track of all the free space, relatively slow (compared to i.e ShelfPacker), but
/// around 90% efficiency.
#[derive(Clone)]
pub struct MaxRectPacker<K> {
    /// Width of the packing area
    width: u32,
    /// Height of the packing area
    height: u32,
    /// Minimum width of a packed item, any free space thinner than that can be ignored
    min_width: u32,
    /// Minimum height of a packed item, any free space shorter than that can be ignored
    min_height: u32,
    /// Used area
    used: u64,
    /// Map of packed items to their respective bounding boxes
    packed: HashMap<K, BoundingBox>,
    /// List of the avalaibles free spaces
    spaces: Vec<BoundingBox>,
    /// Empty most of the time, kept here to reuse the allocation
    spaces_cut: Vec<BoundingBox>,
    /// Optional score function the choose the slot used for an item
    score: Option<fn(BoundingBox, u32, u32) -> i64>,
}

// Heuristics in particular is pretty much a copy paste
#[derive(Debug, Clone, Copy)]
pub enum Heuristics {
    MinX,
    MinY,
    MaxX,
    MaxY,
    BestAreaFit,
    BestShortSideFit,
    BestLongSideFit,
    WorstAreaFit,
    WorstShortSideFit,
    WorstLongSideFit,
}

mod max_rect_packer_heuristics {
    use super::BoundingBox;

    pub fn min_x(rect: BoundingBox, _width: u32, _height: u32) -> i64 {
        -(rect.x1 as i64)
    }
    pub fn min_y(rect: BoundingBox, _width: u32, _height: u32) -> i64 {
        -(rect.y1 as i64)
    }
    pub fn max_x(rect: BoundingBox, _width: u32, _height: u32) -> i64 {
        rect.x2 as i64
    }
    pub fn max_y(rect: BoundingBox, _width: u32, _height: u32) -> i64 {
        rect.y2 as i64
    }
    pub fn best_area_fit(rect: BoundingBox, width: u32, height: u32) -> i64 {
        (width * height) as i64 - rect.area() as i64
    }
    pub fn best_short_side_fit(rect: BoundingBox, width: u32, height: u32) -> i64 {
        (width as i64 - rect.width() as i64).min(height as i64 - rect.height() as i64)
    }
    pub fn best_long_side_fit(rect: BoundingBox, width: u32, height: u32) -> i64 {
        (width as i64 - rect.width() as i64).max(height as i64 - rect.height() as i64)
    }
    pub fn worst_area_fit(rect: BoundingBox, width: u32, height: u32) -> i64 {
        rect.area() as i64 - (width * height) as i64
    }
    pub fn worst_short_side_fit(rect: BoundingBox, width: u32, height: u32) -> i64 {
        (rect.width() as i64 - width as i64).min(rect.height() as i64 - height as i64)
    }
    pub fn worst_long_side_fit(rect: BoundingBox, width: u32, height: u32) -> i64 {
        (rect.width() as i64 - width as i64).max(rect.height() as i64 - height as i64)
    }
}

impl<K: Hash + Eq> MaxRectPacker<K> {
    pub fn new(width: u32, height: u32, heuristics: Option<Heuristics>) -> Self {
        use max_rect_packer_heuristics as heur;

        let score = heuristics.map(|h| match h {
            Heuristics::MinX => heur::min_x,
            Heuristics::MinY => heur::min_y,
            Heuristics::MaxX => heur::max_x,
            Heuristics::MaxY => heur::max_y,
            Heuristics::BestAreaFit => heur::best_area_fit,
            Heuristics::BestShortSideFit => heur::best_short_side_fit,
            Heuristics::BestLongSideFit => heur::best_long_side_fit,
            Heuristics::WorstAreaFit => heur::worst_area_fit,
            Heuristics::WorstShortSideFit => heur::worst_short_side_fit,
            Heuristics::WorstLongSideFit => heur::worst_long_side_fit,
        });

        Self {
            width,
            height,
            used: 0,
            score,
            min_width: 0,
            min_height: 0,
            packed: HashMap::new(),
            spaces: vec![BoundingBox::new_dim(0, 0, width, height)],
            spaces_cut: vec![],
        }
    }
}

impl<K: Hash + Eq> OnlinePacker<K> for MaxRectPacker<K> {
    fn pack(&mut self, item: K, width: u32, height: u32) -> Option<BoundingBox> {
        // Fast path for items with zero area
        if (width == 0 || height == 0) && width <= self.width && height <= self.height {
            // No need to do any packing
            let rect = BoundingBox::new_dim(0, 0, width, height);
            self.packed.insert(item, rect);
            return Some(rect);
        }

        // Pick the slot according the heuristics, chooses the one with the highest score
        let &slot = if let Some(score) = self.score {
            self.spaces
                .iter()
                .filter(|slot| slot.width() >= width && slot.height() >= height)
                .map(|rect| (rect, score(*rect, width, height)))
                .reduce(|(rect, score), (new_rect, new_score)| {
                    if new_score > score {
                        (new_rect, new_score)
                    } else {
                        (rect, score)
                    }
                })
                .map(|(rect, _)| rect)
        } else {
            // no heuristics given, just pick the first one that fits
            self.spaces
                .iter()
                .find(|slot| slot.width() >= width && slot.height() >= height)
        }?;

        // BoundingBox of the item
        let item_box = BoundingBox::new_dim(slot.x1, slot.y1, width, height);

        // Find all the slots that the intersects with, remove them from self.spaces, cut them into
        // max rectangles that don't touch the item, and put those in self.spaces_cut
        self.spaces.retain(|space| {
            if !space.intersects(&item_box) {
                return true;
            }

            self.spaces_cut.extend(
                space
                    .slice(&item_box)
                    .filter(|cut| cut.width() > self.min_width && cut.height() > self.min_height),
            );
            false
        });

        // Prune the redundant rectangles (because of the O(n²) nature of these algorithms, its
        // better to spend some time reducing n than just running with it)

        // Prune the cut rectange with each other, this is O(n²), but only applies to the cut
        // rectangles (around 1 to 30)
        {
            // No iterators here because we are modifying the vector in place, and ownership rules
            // make this painful
            let mut i = 0;
            while i < self.spaces_cut.len() {
                let mut j = i + 1;
                while j < self.spaces_cut.len() {
                    if self.spaces_cut[j].contains(&self.spaces_cut[i]) {
                        self.spaces_cut.remove(i);
                        i = i.wrapping_sub(1);
                        break;
                    }

                    if self.spaces_cut[i].contains(&self.spaces_cut[j]) {
                        self.spaces_cut.remove(j);
                        j = j.wrapping_sub(1);
                    }
                    j = j.wrapping_add(1);
                }
                i = i.wrapping_add(1);
            }
        }

        // Merge back the cut rectangles with the others, each cut rectange is tested against all
        // other rectangles. Splitting the cut and uncut rectangles lets us avoid testing
        // untouched rectanges against all other (goes from O(n²), with n (~500) the number of rectangles,
        // to O(c² + cu) with c (~30) and u (~500) the number of cut and uncut rectangles
        for space in self.spaces_cut.drain(..) {
            // Iterators are a lot nicer
            if !self.spaces.iter().any(|s| s.contains(&space)) {
                self.spaces.push(space);
            }
        }

        self.packed.insert(item, item_box);
        self.used += item_box.area() as u64;
        Some(item_box)
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn get_loc(&self, item: &K) -> Option<BoundingBox> {
        self.packed.get(item).copied()
    }

    fn usage(&self) -> f32 {
        self.used as f32 / (self.width as f32 * self.height as f32)
    }
    fn set_min_size(&mut self, min_width: u32, min_height: u32) {
        self.min_width = min_width;
        self.min_height = min_height;
    }
}

/// Packer receiving items one by one in a sequence, cannot move a previously placed item
pub trait OnlinePacker<K> {
    /// Pack an item of a certain size, returns Some(BoundingBox) if the item could be packed, None
    /// otherwise
    fn pack(&mut self, item: K, width: u32, height: u32) -> Option<BoundingBox>;
    /// Get the location of a packed item
    fn get_loc(&self, item: &K) -> Option<BoundingBox>;
    /// The width of the packing area
    fn width(&self) -> u32;
    /// The height of the packing area
    fn height(&self) -> u32;
    /// Usage factor used_area / packing_area, [0-1]
    fn usage(&self) -> f32;
    /// Set the minimum item size for the packer, some packer use this information for optimizations
    fn set_min_size(&mut self, _min_width: u32, _min_height: u32) {}
}

pub trait LayeredOnlinePacker<K> {
    /// Pack an item of a certain size, returns Some((BoundingBox, layers)) if the item could be packed, None
    /// otherwise.
    fn pack(&mut self, item: K, width: u32, height: u32) -> Option<(BoundingBox, u32)>;
    /// Pack an item of a certain size onto a particular layer, returns Some(BoundingBox) if the
    /// item could be packed, None otherwise.
    fn pack_layer(&mut self, item: K, width: u32, height: u32, layer: u32) -> Option<BoundingBox>;
    /// Get the location of a packed item
    fn get_loc(&self, item: &K) -> Option<(BoundingBox, u32)>;
    /// The width of the packing area
    fn width(&self) -> u32;
    /// The height of the packing area
    fn height(&self) -> u32;
    /// The number of availables layers
    fn layers(&self) -> u32;
    /// Add layers to the packer
    fn add_layers(&mut self, layers: u32);
    /// Usage factor used_area / packing_area, [0-1]
    fn usage(&self) -> f32;
    /// Set the minimum item size for the packer, some packer use this information for optimizations
    fn set_min_size(&mut self, _min_width: u32, _min_height: u32) {}
}

pub struct Layered<P, K>
where
    P: OnlinePacker<K>,
{
    layers: Vec<P>,
    width: u32,
    height: u32,
    builder: Box<dyn Fn(u32, u32) -> P>,
    _phantom: PhantomData<K>,
}

impl<K, P: OnlinePacker<K>> Layered<P, K> {
    pub fn new<F>(width: u32, height: u32, builder: F) -> Self
    where
        F: Fn(u32, u32) -> P + 'static,
    {
        let layers = vec![builder(width, height)];
        Self {
            width,
            height,
            builder: Box::new(builder),
            layers,
            _phantom: PhantomData,
        }
    }
}

impl<K: Clone, P: OnlinePacker<K>> LayeredOnlinePacker<K> for Layered<P, K> {
    fn pack(&mut self, item: K, width: u32, height: u32) -> Option<(BoundingBox, u32)> {
        self.layers
            .iter_mut()
            .enumerate()
            .find_map(|(index, layer)| {
                Some((layer.pack(item.clone(), width, height)?, index as u32))
            })
    }

    fn pack_layer(&mut self, item: K, width: u32, height: u32, layer: u32) -> Option<BoundingBox> {
        self.layers
            .get_mut(layer as usize)?
            .pack(item, width, height)
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn layers(&self) -> u32 {
        self.layers.len() as u32
    }

    fn usage(&self) -> f32 {
        self.layers.iter().map(|layer| layer.usage()).sum::<f32>() / self.layers() as f32
    }

    fn get_loc(&self, item: &K) -> Option<(BoundingBox, u32)> {
        self.layers
            .iter()
            .enumerate()
            .find_map(|(index, layer)| Some((layer.get_loc(item)?, index as u32)))
    }

    fn add_layers(&mut self, layers: u32) {
        self.layers.extend(
            std::iter::repeat_with(|| (self.builder)(self.width, self.height))
                .take(layers as usize),
        );
    }
}
