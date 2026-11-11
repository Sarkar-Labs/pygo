package tensor

import "fmt"

func Fused1599Log2MatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.GELU()
	return r, nil
}
func Fused1599Log2MatmulGeluName() string { return "log2_matmul_gelu" }
