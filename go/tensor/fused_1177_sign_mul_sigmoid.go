package tensor

import "fmt"

func Fused1177SignMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Sigmoid()
	return r, nil
}
func Fused1177SignMulSigmoidName() string { return "sign_mul_sigmoid" }
