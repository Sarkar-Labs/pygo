package tensor

import "fmt"

func Fused0521SqrtSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.SiLU()
	return r, nil
}
func Fused0521SqrtSubSiluName() string { return "sqrt_sub_silu" }
