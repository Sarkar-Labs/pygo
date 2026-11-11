package tensor

import "fmt"

func Fused1321GeluSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Sign()
	return r, nil
}
func Fused1321GeluSubSignName() string { return "gelu_sub_sign" }
