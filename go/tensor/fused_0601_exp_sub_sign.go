package tensor

import "fmt"

func Fused0601ExpSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Sign()
	return r, nil
}
func Fused0601ExpSubSignName() string { return "exp_sub_sign" }
