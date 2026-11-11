package tensor

import "fmt"

func Fused1636Log10SubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Abs()
	return r, nil
}
func Fused1636Log10SubAbsName() string { return "log10_sub_abs" }
