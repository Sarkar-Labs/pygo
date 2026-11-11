package tensor

import "fmt"

func Fused1441NegSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.ReLU()
	return r, nil
}
func Fused1441NegSubReluName() string { return "neg_sub_relu" }
