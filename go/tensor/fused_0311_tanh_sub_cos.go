package tensor

import "fmt"

func Fused0311TanhSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Cos()
	return r, nil
}
func Fused0311TanhSubCosName() string { return "tanh_sub_cos" }
