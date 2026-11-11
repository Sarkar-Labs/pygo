package tensor

import "fmt"

func Fused1275GeluAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Abs()
	return r, nil
}
func Fused1275GeluAddAbsName() string { return "gelu_add_abs" }
