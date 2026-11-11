package tensor

import "fmt"

func Fused1594Log2MatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Sign()
	return r, nil
}
func Fused1594Log2MatmulSignName() string { return "log2_matmul_sign" }
