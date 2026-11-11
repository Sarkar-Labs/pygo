package tensor

import "fmt"

func Fused1204SignMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Log()
	return r, nil
}
func Fused1204SignMatmulLogName() string { return "sign_matmul_log" }
