package tensor

import "fmt"

func Fused0380AbsAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Sqrt()
	return r, nil
}
func Fused0380AbsAddSqrtName() string { return "abs_add_sqrt" }
