package tensor

import "fmt"

func Fused1366SiluSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Abs()
	return r, nil
}
func Fused1366SiluSubAbsName() string { return "silu_sub_abs" }
