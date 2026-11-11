package tensor

import "fmt"

func Fused0761SinSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Cos()
	return r, nil
}
func Fused0761SinSubCosName() string { return "sin_sub_cos" }
