package tensor

import "fmt"

func Fused0277TanhMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Sigmoid()
	return r, nil
}
func Fused0277TanhMulSigmoidName() string { return "tanh_mul_sigmoid" }
