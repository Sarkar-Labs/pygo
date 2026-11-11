package tensor

import "fmt"

func Fused0694LogMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Sign()
	return r, nil
}
func Fused0694LogMatmulSignName() string { return "log_matmul_sign" }
