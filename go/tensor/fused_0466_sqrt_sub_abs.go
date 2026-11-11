package tensor

import "fmt"

func Fused0466SqrtSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Abs()
	return r, nil
}
func Fused0466SqrtSubAbsName() string { return "sqrt_sub_abs" }
