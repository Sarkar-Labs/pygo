package tensor

import "fmt"

func Fused0071SiluSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	return r, nil
}
func Fused0071SiluSubName() string { return "silu_sub" }
