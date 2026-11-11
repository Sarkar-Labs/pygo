package tensor

import "fmt"

func Fused1240SignAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.SiLU()
	return r, nil
}
func Fused1240SignAddSiluName() string { return "sign_add_silu" }
