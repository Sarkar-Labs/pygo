package tensor

import "fmt"

func Fused1501NegSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Sign()
	return r, nil
}
func Fused1501NegSubSignName() string { return "neg_sub_sign" }
