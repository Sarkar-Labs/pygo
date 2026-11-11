package tensor

import "fmt"

func Fused1174SignMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.ReLU()
	return r, nil
}
func Fused1174SignMatmulReluName() string { return "sign_matmul_relu" }
