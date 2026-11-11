package tensor

import "fmt"

func Fused1231SignSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Sign()
	return r, nil
}
func Fused1231SignSubSignName() string { return "sign_sub_sign" }
