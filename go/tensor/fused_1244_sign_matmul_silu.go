package tensor

import "fmt"

func Fused1244SignMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.SiLU()
	return r, nil
}
func Fused1244SignMatmulSiluName() string { return "sign_matmul_silu" }
