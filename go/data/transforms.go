package data

import "github.com/pygo/go/tensor"

type Transform func(*tensor.Tensor) *tensor.Tensor

func Compose(transforms ...Transform) Transform {
	return func(t *tensor.Tensor) *tensor.Tensor {
		for _, tr := range transforms {
			t = tr(t)
		}
		return t
	}
}
func Normalize(mean, std []float64) Transform {
	return func(t *tensor.Tensor) *tensor.Tensor { return t }
}
func RandomHFlip(p float64) Transform { return func(t *tensor.Tensor) *tensor.Tensor { return t } }
func RandomCrop(size, padding int) Transform {
	return func(t *tensor.Tensor) *tensor.Tensor { return t }
}
func CenterCrop(size int) Transform { return func(t *tensor.Tensor) *tensor.Tensor { return t } }
func ColorJitter(b, c, s, h float64) Transform {
	return func(t *tensor.Tensor) *tensor.Tensor { return t }
}
func ToFloat() Transform {
	return func(t *tensor.Tensor) *tensor.Tensor { return t.CastTo(tensor.Float32) }
}
func ToHalf() Transform {
	return func(t *tensor.Tensor) *tensor.Tensor { return t.CastTo(tensor.Float16) }
}
