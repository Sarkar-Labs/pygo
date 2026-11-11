package tensor

import "fmt"

func Fused1254SignMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Log2()
	return r, nil
}
func Fused1254SignMatmulLog2Name() string { return "sign_matmul_log2" }
