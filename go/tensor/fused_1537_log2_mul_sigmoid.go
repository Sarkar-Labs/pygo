package tensor

import "fmt"

func Fused1537Log2MulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Sigmoid()
	return r, nil
}
func Fused1537Log2MulSigmoidName() string { return "log2_mul_sigmoid" }
