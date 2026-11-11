package tensor

import "fmt"

func Fused1635Log10AddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Abs()
	return r, nil
}
func Fused1635Log10AddAbsName() string { return "log10_add_abs" }
