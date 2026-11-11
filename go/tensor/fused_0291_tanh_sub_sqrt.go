package tensor

import "fmt"

func Fused0291TanhSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Sqrt()
	return r, nil
}
func Fused0291TanhSubSqrtName() string { return "tanh_sub_sqrt" }
