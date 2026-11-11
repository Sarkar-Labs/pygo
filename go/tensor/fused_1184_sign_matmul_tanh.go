package tensor

import "fmt"

func Fused1184SignMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Tanh()
	return r, nil
}
func Fused1184SignMatmulTanhName() string { return "sign_matmul_tanh" }
