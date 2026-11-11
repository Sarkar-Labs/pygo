package tensor

import "fmt"

func Fused1241SignSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.SiLU()
	return r, nil
}
func Fused1241SignSubSiluName() string { return "sign_sub_silu" }
