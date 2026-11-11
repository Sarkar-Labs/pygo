package tensor

import "fmt"

func Fused1371SiluSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Sqrt()
	return r, nil
}
func Fused1371SiluSubSqrtName() string { return "silu_sub_sqrt" }
