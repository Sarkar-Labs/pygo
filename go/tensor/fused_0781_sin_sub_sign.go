package tensor

import "fmt"

func Fused0781SinSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Sign()
	return r, nil
}
func Fused0781SinSubSignName() string { return "sin_sub_sign" }
