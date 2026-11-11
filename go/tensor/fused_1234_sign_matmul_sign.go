package tensor

import "fmt"

func Fused1234SignMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Sign()
	return r, nil
}
func Fused1234SignMatmulSignName() string { return "sign_matmul_sign" }
