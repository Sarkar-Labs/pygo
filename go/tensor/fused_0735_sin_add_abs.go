package tensor

import "fmt"

func Fused0735SinAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Abs()
	return r, nil
}
func Fused0735SinAddAbsName() string { return "sin_add_abs" }
