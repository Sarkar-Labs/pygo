package tensor

import "fmt"

func Fused0085Log10Add(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	return r, nil
}
func Fused0085Log10AddName() string { return "log10_add" }
