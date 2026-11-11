package tensor

import "fmt"

func Fused1179SignMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Sigmoid()
	return r, nil
}
func Fused1179SignMatmulSigmoidName() string { return "sign_matmul_sigmoid" }
