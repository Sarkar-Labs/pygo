package tensor

import "fmt"

func Fused1442NegMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.ReLU()
	return r, nil
}
func Fused1442NegMulReluName() string { return "neg_mul_relu" }
