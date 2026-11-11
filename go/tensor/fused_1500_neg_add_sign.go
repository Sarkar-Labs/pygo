package tensor

import "fmt"

func Fused1500NegAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Sign()
	return r, nil
}
func Fused1500NegAddSignName() string { return "neg_add_sign" }
