package tensor

import "fmt"

func Fused1051CeilSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Sign()
	return r, nil
}
func Fused1051CeilSubSignName() string { return "ceil_sub_sign" }
