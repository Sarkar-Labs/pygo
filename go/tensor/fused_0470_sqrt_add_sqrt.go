package tensor

import "fmt"

func Fused0470SqrtAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Sqrt()
	return r, nil
}
func Fused0470SqrtAddSqrtName() string { return "sqrt_add_sqrt" }
