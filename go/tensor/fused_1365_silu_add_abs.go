package tensor

import "fmt"

func Fused1365SiluAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Abs()
	return r, nil
}
func Fused1365SiluAddAbsName() string { return "silu_add_abs" }
