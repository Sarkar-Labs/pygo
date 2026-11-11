package tensor

import "fmt"

func Fused0465SqrtAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Abs()
	return r, nil
}
func Fused0465SqrtAddAbsName() string { return "sqrt_add_abs" }
