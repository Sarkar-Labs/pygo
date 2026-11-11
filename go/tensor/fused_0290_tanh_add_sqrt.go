package tensor

import "fmt"

func Fused0290TanhAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Sqrt()
	return r, nil
}
func Fused0290TanhAddSqrtName() string { return "tanh_add_sqrt" }
