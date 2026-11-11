package tensor

import "fmt"

func Fused1402SiluMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Ceil()
	return r, nil
}
func Fused1402SiluMulCeilName() string { return "silu_mul_ceil" }
