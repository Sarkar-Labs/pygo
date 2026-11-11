package tensor

import "fmt"

func Fused1239SignMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.GELU()
	return r, nil
}
func Fused1239SignMatmulGeluName() string { return "sign_matmul_gelu" }
