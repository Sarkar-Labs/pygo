package tensor

import "fmt"

func Fused0076NegSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	return r, nil
}
func Fused0076NegSubName() string { return "neg_sub" }
