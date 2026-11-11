package tensor

import "fmt"

func Fused1411SiluSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Sign()
	return r, nil
}
func Fused1411SiluSubSignName() string { return "silu_sub_sign" }
