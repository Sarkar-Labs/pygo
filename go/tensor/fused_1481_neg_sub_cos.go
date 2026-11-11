package tensor

import "fmt"

func Fused1481NegSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Cos()
	return r, nil
}
func Fused1481NegSubCosName() string { return "neg_sub_cos" }
