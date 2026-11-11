package tensor

import "fmt"

func Fused0736SinSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Abs()
	return r, nil
}
func Fused0736SinSubAbsName() string { return "sin_sub_abs" }
