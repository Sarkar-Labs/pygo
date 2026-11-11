package tensor

import "fmt"

func Fused0784SinMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Sign()
	return r, nil
}
func Fused0784SinMatmulSignName() string { return "sin_matmul_sign" }
